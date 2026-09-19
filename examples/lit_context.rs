// How much literal context is worth: order-0 entropy of the ultra parse's
// literals against their entropy conditioned on cheap contexts, per file
// and for the corpus, in KB of literal section. Contexts: the previous
// output byte's class (top 2, 3, 4 bits), the byte before the run, and
// the literal 8 back in literal order (the same decode stream).
use glyd::v7_encode::Sequence;
use glyd::v7_ultra::{find_sequences_ultra, UltraState};

fn ent(h: &[u32]) -> f64 {
    let t: u32 = h.iter().sum();
    if t == 0 { return 0.0; }
    h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (t as f64 / c as f64).log2()).sum()
}
/// Conditional entropy with `nctx` contexts, plus a 90-byte table per
/// context used (the price of carrying the tables per block).
fn cond(hist: &[Vec<u32>], per_table_bits: f64) -> f64 {
    hist.iter().map(|h| if h.iter().any(|&c| c > 0) { ent(h) + per_table_bits } else { 0.0 }).sum()
}

fn main() {
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    let ext = std::path::Path::new("corpus/ext");
    if ext.is_dir() { for e in std::fs::read_dir(ext).unwrap().filter_map(|e| e.ok()) { if e.path().is_file() { files.push(e.path()); } } }
    files.sort();
    let mut st = UltraState::new();
    let names = ["order-0", "prev top2", "prev top3", "prev top4", "prev byte (256)", "lit-8 top3", "first-of-run split"];
    let mut tot = [0f64; 7];
    println!("{:22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}", "file (literal KB)", "o0", "p2", "p3", "p4", "p8", "l8t3", "run");
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        if d.len() > 300 << 20 { continue; }
        st.clear(d.len());
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let mut sums = [0f64; 7];
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            // Walk the sequences to find each literal's output position.
            let mut h0 = vec![0u32; 256];
            let mut h2 = vec![vec![0u32; 256]; 4];
            let mut h3 = vec![vec![0u32; 256]; 8];
            let mut h4 = vec![vec![0u32; 256]; 16];
            let mut h8 = vec![vec![0u32; 256]; 256];
            let mut hl8 = vec![vec![0u32; 256]; 8];
            let mut hrun = vec![vec![0u32; 256]; 2];
            let (mut pos, mut li) = (off, 0usize);
            for q in &seqs {
                for j in 0..q.lit_len as usize {
                    let b = d[pos + j] as usize;
                    let prev = if pos + j > 0 { d[pos + j - 1] as usize } else { 0 };
                    h0[b] += 1;
                    h2[prev >> 6][b] += 1;
                    h3[prev >> 5][b] += 1;
                    h4[prev >> 4][b] += 1;
                    h8[prev][b] += 1;
                    let l8 = if li >= 8 { lits[li - 8] as usize } else { 0 };
                    hl8[l8 >> 5][b] += 1;
                    hrun[(j == 0) as usize][b] += 1;
                    li += 1;
                }
                pos += (q.lit_len + q.match_len) as usize;
            }
            let tb = 90.0 * 8.0;
            sums[0] += ent(&h0) + tb;
            sums[1] += cond(&h2, tb); sums[2] += cond(&h3, tb); sums[3] += cond(&h4, tb);
            sums[4] += cond(&h8, tb); sums[5] += cond(&hl8, tb); sums[6] += cond(&hrun, tb);
            off += len;
        }
        let kb = |b: f64| b / 8192.0;
        println!("{:22} {:9.0} {:8.1}% {:8.1}% {:8.1}% {:8.1}% {:8.1}% {:8.1}%", format!("{name} ({:.0})", kb(sums[0])), kb(sums[0]),
            100.0 * (sums[1] / sums[0] - 1.0), 100.0 * (sums[2] / sums[0] - 1.0), 100.0 * (sums[3] / sums[0] - 1.0), 100.0 * (sums[4] / sums[0] - 1.0), 100.0 * (sums[5] / sums[0] - 1.0), 100.0 * (sums[6] / sums[0] - 1.0));
        for i in 0..7 { tot[i] += sums[i]; }
    }
    println!("{:22} {:9.0} {:8.1}% {:8.1}% {:8.1}% {:8.1}% {:8.1}% {:8.1}%", "total", tot[0] / 8192.0,
        100.0 * (tot[1] / tot[0] - 1.0), 100.0 * (tot[2] / tot[0] - 1.0), 100.0 * (tot[3] / tot[0] - 1.0), 100.0 * (tot[4] / tot[0] - 1.0), 100.0 * (tot[5] / tot[0] - 1.0), 100.0 * (tot[6] / tot[0] - 1.0));
    let _ = names;
}
