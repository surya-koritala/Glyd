// The v7 parse alone on Silesia: `find_sequences_dfast` timed by itself in
// 256 KB blocks (tables carried across blocks as `compress_into_max` does),
// next to the end-to-end `compress_into_max` time and ratio, so a parse
// change is read as parse ns/byte, total ns/byte and ratio in one run.
//
// Usage: cargo run --release --example v7_parse [runs=5] [min_s=0.2]
//        cargo run --release --example v7_parse -- --loop [secs=8]
//           (parse only, for `xctrace record --template 'Time Profiler'`)
use glyd::v7_encode::{find_sequences_dfast, DfastTables, EncScratch};
use std::time::Instant;

const BLOCK: usize = 256 * 1024;

fn timed<F: FnMut()>(runs: usize, min_s: f64, mut op: F) -> f64 {
    let mut v = Vec::new();
    for _ in 0..runs {
        op();
        let t = Instant::now();
        let mut n = 0;
        loop {
            op();
            n += 1;
            let e = t.elapsed().as_secs_f64();
            if e >= min_s && n >= 2 {
                v.push(e / n as f64);
                break;
            }
        }
    }
    // The minimum: robust to the other agents' load on the machine.
    v.into_iter().fold(f64::INFINITY, f64::min)
}

fn parse_all(d: &[u8], t: &mut DfastTables, seqs: &mut Vec<glyd::v7_encode::Sequence>, lits: &mut Vec<u8>, codes: &mut EncScratch) -> usize {
    let mut n = 0;
    let mut off = 0;
    while off < d.len() {
        let len = (d.len() - off).min(BLOCK);
        seqs.clear();
        lits.clear();
        let mut reps = [1u32, 4, 8];
        find_sequences_dfast(d, off, len, t, &mut reps, seqs, lits, codes);
        n += seqs.len();
        off += len;
    }
    n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let files = ["dickens", "mozilla", "mr", "nci", "ooffice", "osdb", "reymont", "samba", "sao", "webster", "xml", "x-ray"];
    let dir = std::path::Path::new("corpus");
    let data: Vec<(&str, Vec<u8>)> = files.iter().filter_map(|f| std::fs::read(dir.join(f)).ok().map(|d| (*f, d))).collect();
    let mut t = DfastTables::new();
    let (mut seqs, mut lits, mut codes) = (Vec::new(), Vec::new(), EncScratch::new());

    if args.iter().any(|a| a == "--loop") {
        let secs: f64 = args.last().and_then(|s| s.parse().ok()).unwrap_or(8.0);
        let start = Instant::now();
        let mut n = 0usize;
        while start.elapsed().as_secs_f64() < secs {
            for (_, d) in &data {
                n += parse_all(d, &mut t, &mut seqs, &mut lits, &mut codes);
            }
        }
        std::hint::black_box(n);
        return;
    }

    let runs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5);
    let min_s: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let (mut tot_in, mut tot_out, mut tot_pt, mut tot_ct, mut tot_seq) = (0usize, 0usize, 0.0f64, 0.0f64, 0usize);
    println!("{:<8} {:>8} {:>8} {:>8} {:>10} {:>8}", "file", "parse", "total", "ratio", "seqs", "B/seq");
    for (f, d) in &data {
        let n_seq = parse_all(d, &mut t, &mut seqs, &mut lits, &mut codes);
        let pt = timed(runs, min_s, || {
            parse_all(d, &mut t, &mut seqs, &mut lits, &mut codes);
        });
        let mut out = Vec::with_capacity(d.len());
        glyd::compress_into_max(d, &mut out);
        let ct = timed(runs, min_s, || {
            out.clear();
            glyd::compress_into_max(d, &mut out);
        });
        let ns = 1e9 / d.len() as f64;
        println!("{:<8} {:>8.3} {:>8.3} {:>8.4} {:>10} {:>8.1}", f, pt * ns, ct * ns, d.len() as f64 / out.len() as f64, n_seq, d.len() as f64 / n_seq as f64);
        tot_in += d.len();
        tot_out += out.len();
        tot_pt += pt;
        tot_ct += ct;
        tot_seq += n_seq;
    }
    let ns = 1e9 / tot_in as f64;
    let gb = 1024.0 * 1024.0 * 1024.0;
    println!(
        "{:<8} {:>8.3} {:>8.3} {:>8.4} {:>10} {:>8.1}   comp {:.3} GB/s",
        "total",
        tot_pt * ns,
        tot_ct * ns,
        tot_in as f64 / tot_out as f64,
        tot_seq,
        tot_in as f64 / tot_seq as f64,
        (tot_in as f64 / gb) / tot_ct
    );
}
