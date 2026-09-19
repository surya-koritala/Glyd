// How the ultra parse uses repeat offsets, per file: rep-code share, and
// what zstd's ll0 rule would touch (matches right after a match at rep0,
// and at rep0 - 1).
use glyd::v7_encode::Sequence;
use glyd::v7_ultra::{find_sequences_ultra, UltraState};

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let mut st = UltraState::new();
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        st.clear(d.len());
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let (mut n, mut rep, mut ll0, mut ll0_rep0, mut ll0_rep0m1, mut rep0m1) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            let mut r = [1u32, 4, 8];
            for q in &seqs {
                if q.match_len == 0 { continue; }
                n += 1;
                let is_rep = r.contains(&q.offset);
                if is_rep { rep += 1; }
                if q.offset + 1 == r[0] { rep0m1 += 1; }
                if q.lit_len == 0 {
                    ll0 += 1;
                    if q.offset == r[0] { ll0_rep0 += 1; }
                    if q.offset + 1 == r[0] { ll0_rep0m1 += 1; }
                }
                r = if q.offset == r[0] { r } else if q.offset == r[1] { [r[1], r[0], r[2]] } else if q.offset == r[2] { [r[2], r[0], r[1]] } else { [q.offset, r[0], r[1]] };
            }
            off += len;
        }
        println!("{name:10} {n:8} matches: rep {:5.1}%, ll0 {:5.1}% (of which rep0 {:5.1}%, rep0-1 {:5.1}%); rep0-1 overall {:5.2}%",
            100.0 * rep as f64 / n as f64, 100.0 * ll0 as f64 / n as f64, 100.0 * ll0_rep0 as f64 / ll0.max(1) as f64, 100.0 * ll0_rep0m1 as f64 / ll0.max(1) as f64, 100.0 * rep0m1 as f64 / n as f64);
    }
}
