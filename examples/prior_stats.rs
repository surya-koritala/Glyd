// Code histograms of the max level's parse over Silesia, normalised to
// 4096 per table: the ultra parse's starting prices.
use glyd::v7_encode::{find_sequences_dfast, DfastTables, EncScratch, Sequence};
use glyd::v7_format::{ll_code, ml_code, LL_SYMBOLS, ML_SYMBOLS, OFF_SYMBOLS};

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let (mut hll, mut hml, mut hoff) = ([0u64; LL_SYMBOLS], [0u64; ML_SYMBOLS], [0u64; OFF_SYMBOLS]);
    let mut t = DfastTables::new();
    let mut codes = EncScratch::new();
    let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
    for p in files {
        let d = std::fs::read(&p).unwrap();
        t.clear();
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            seqs.clear(); lits.clear();
            let mut reps = [1u32, 4, 8];
            find_sequences_dfast(&d, off, len, &mut t, &mut reps, &mut seqs, &mut lits, &mut codes);
            let mut r = [1u32, 4, 8];
            for q in &seqs {
                hll[ll_code(q.lit_len).0 as usize] += 1;
                if q.match_len == 0 { continue; }
                hml[ml_code(q.match_len).0 as usize] += 1;
                let c = if q.offset == r[0] { 0 } else if q.offset == r[1] { 1 } else if q.offset == r[2] { 2 } else { 3 + (31 - q.offset.leading_zeros()) as usize };
                hoff[c] += 1;
                r = if q.offset == r[0] { r } else if q.offset == r[1] { [r[1], r[0], r[2]] } else if q.offset == r[2] { [r[2], r[0], r[1]] } else { [q.offset, r[0], r[1]] };
            }
            off += len;
        }
    }
    for (name, h) in [("LL", &hll[..]), ("ML", &hml[..]), ("OFF", &hoff[..])] {
        let t: u64 = h.iter().sum();
        let v: Vec<u32> = h.iter().map(|&c| ((c as f64 * 4096.0 / t as f64).round() as u32).max(1)).collect();
        println!("const PRIOR_{name}: [u32; {}] = {:?};", v.len(), v);
    }
}
