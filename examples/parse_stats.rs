// Scratch: what the max level's parse finds on a file (one block
// stream): sequences, literal bytes, match lengths and offsets, against
// the ultra parse on the same bytes.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(&f).unwrap();
    let n = d.len().min(64 << 20);
    let d = &d[..n];
    let far = glyd::ldm::Matches { list: Vec::new() };
    let mut tables = glyd::v7_encode::DfastTables::new();
    tables.clear_for(n);
    let mut scratch = glyd::v7_encode::EncScratch::new();
    let (mut seqs, mut lits) = (Vec::new(), Vec::new());
    let mut offset = 0usize;
    while offset < n {
        let len = (n - offset).min(128 << 10);
        let mut reps = [1u32, 4, 8];
        glyd::v7_encode::find_sequences_dfast_far(d, offset, len, &mut tables, &far, &mut reps, &mut seqs, &mut lits, &mut scratch);
        offset += len;
    }
    report("dfast", n, &seqs, lits.len());
    if let Ok(out) = std::env::var("GLYD_SEQ_OUT") {
        // litLen, matchLen, offset as u32 triples; the input's first n bytes beside it
        let mut b = Vec::with_capacity(seqs.len() * 12);
        for q in &seqs {
            b.extend_from_slice(&q.lit_len.to_le_bytes());
            b.extend_from_slice(&q.match_len.to_le_bytes());
            b.extend_from_slice(&q.offset.to_le_bytes());
        }
        std::fs::write(&out, b).unwrap();
        std::fs::write(format!("{out}.input"), d).unwrap();
        return;
    }
    let mut u = glyd::v7_ultra::UltraState::new();
    u.clear(n);
    let (mut seqs, mut lits) = (Vec::new(), Vec::new());
    let mut offset = 0usize;
    while offset < n {
        let len = (n - offset).min(128 << 10);
        glyd::v7_ultra::find_sequences_ultra_far(d, offset, len, &mut u, [1, 4, 8], &far, &mut seqs, &mut lits);
        offset += len;
    }
    report("ultra", n, &seqs, lits.len());
}

fn report(name: &str, n: usize, seqs: &[glyd::v7_encode::Sequence], lits: usize) {
    let matches: Vec<&glyd::v7_encode::Sequence> = seqs.iter().filter(|q| q.match_len > 0).collect();
    let mlen: u64 = matches.iter().map(|q| q.match_len as u64).sum();
    let mut hist = [0usize; 6];
    let mut lhist = [0usize; 5];
    for q in &matches {
        let o = q.offset as usize;
        hist[if o < 64 { 0 } else if o < 4096 { 1 } else if o < 65536 { 2 } else if o < 1 << 21 { 3 } else if o < 1 << 23 { 4 } else { 5 }] += 1;
        let l = q.match_len as usize;
        lhist[if l < 6 { 0 } else if l < 10 { 1 } else if l < 20 { 2 } else if l < 50 { 3 } else { 4 }] += 1;
    }
    println!("{name}: {} MB, {} matches ({} per KB), literals {:.1}%, mean match {:.1}; offsets <64 {:.0}% <4K {:.0}% <64K {:.0}% <2M {:.0}% <8M {:.0}% more {:.0}%; lengths <6 {:.0}% <10 {:.0}% <20 {:.0}% <50 {:.0}% more {:.0}%",
        n >> 20, matches.len(), matches.len() * 1024 / n, 100.0 * lits as f64 / n as f64, mlen as f64 / matches.len().max(1) as f64,
        100.0 * hist[0] as f64 / matches.len() as f64, 100.0 * hist[1] as f64 / matches.len() as f64, 100.0 * hist[2] as f64 / matches.len() as f64, 100.0 * hist[3] as f64 / matches.len() as f64, 100.0 * hist[4] as f64 / matches.len() as f64, 100.0 * hist[5] as f64 / matches.len() as f64,
        100.0 * lhist[0] as f64 / matches.len() as f64, 100.0 * lhist[1] as f64 / matches.len() as f64, 100.0 * lhist[2] as f64 / matches.len() as f64, 100.0 * lhist[3] as f64 / matches.len() as f64, 100.0 * lhist[4] as f64 / matches.len() as f64);
}
