// Estimated sequence-section bits for the ultra parse under the current
// length codes (direct below 16, then log2 buckets) and under zstd's
// (direct to 34/15, then 1,1,2,2,3,3... extra bits): code entropy plus
// extra bits, per file and in total.
use glyd::v7_encode::Sequence;
use glyd::v7_ultra::{find_sequences_ultra, UltraState};

fn entropy_bits(h: &[u64]) -> f64 {
    let t: u64 = h.iter().sum();
    h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * ((t as f64) / c as f64).log2()).sum()
}
// zstd match-length code: 3..=34 direct, then buckets.
fn zstd_ml(len: u32) -> (usize, u32) {
    let v = len - 3;
    if v < 32 { return (v as usize, 0); }
    let k = 31 - v.leading_zeros();
    // zstd: 35-36 (1 bit), 37-40 (2), 41-48 (3), 49-64 (4), 65-96 (5)... approximately two codes per power of two
    let half = if v & (1 << (k - 1)) != 0 { 1 } else { 0 };
    (32 + (k as usize - 5) * 2 + half, k - 1)
}
fn zstd_ll(ll: u32) -> (usize, u32) {
    if ll < 16 { return (ll as usize, 0); }
    let k = 31 - ll.leading_zeros();
    let half = if ll & (1 << (k - 1)) != 0 { 1 } else { 0 };
    (16 + (k as usize - 4) * 2 + half, k - 1)
}
fn cur_len(v: u32) -> (usize, u32) { if v < 16 { (v as usize, 0) } else { let k = 31 - v.leading_zeros(); (12 + k as usize, k) } }

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let mut st = UltraState::new();
    let (mut tot_cur, mut tot_z) = (0f64, 0f64);
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        st.clear(d.len());
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let (mut hc_ll, mut hc_ml, mut hz_ll, mut hz_ml) = (vec![0u64; 64], vec![0u64; 64], vec![0u64; 64], vec![0u64; 64]);
        let (mut xc, mut xz) = (0u64, 0u64);
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            for q in &seqs {
                let (c, x) = cur_len(q.lit_len); hc_ll[c] += 1; xc += x as u64;
                let (c, x) = zstd_ll(q.lit_len); hz_ll[c] += 1; xz += x as u64;
                if q.match_len == 0 { continue; }
                let (c, x) = cur_len(q.match_len - 3); hc_ml[c] += 1; xc += x as u64;
                let (c, x) = zstd_ml(q.match_len); hz_ml[c] += 1; xz += x as u64;
            }
            off += len;
        }
        let cur = entropy_bits(&hc_ll) + entropy_bits(&hc_ml) + xc as f64;
        let z = entropy_bits(&hz_ll) + entropy_bits(&hz_ml) + xz as f64;
        println!("{name:12} current {:9.0} KB  zstd-like {:9.0} KB  ({:+.1}%)", cur / 8192.0, z / 8192.0, 100.0 * (z / cur - 1.0));
        tot_cur += cur; tot_z += z;
    }
    println!("total        current {:9.0} KB  zstd-like {:9.0} KB  ({:+.1}%)", tot_cur / 8192.0, tot_z / 8192.0, 100.0 * (tot_z / tot_cur - 1.0));
}
