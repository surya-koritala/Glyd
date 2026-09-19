// Sequence counts and literal shares of the max and ultra parses on one
// file, per block, next to zstd's levels in a 2 MB window.
use glyd::v7_encode::{find_sequences_dfast, DfastTables, EncScratch, Sequence};
use glyd::v7_ultra::{find_sequences_ultra, UltraState};
use std::io::Write;

fn main() {
    let f = std::env::args().nth(1).expect("file");
    let d = std::fs::read(format!("corpus/{f}")).unwrap();
    let mut t = DfastTables::new();
    let mut codes = EncScratch::new();
    let mut st = UltraState::new();
    let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
    for (name, ultra) in [("max", false), ("ultra", true)] {
        let (mut ns, mut nl, mut nm3, mut rep) = (0usize, 0usize, 0usize, 0usize);
        let mut off = 0;
        t.clear();
        st.clear(d.len());
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            seqs.clear();
            lits.clear();
            let mut reps = [1u32, 4, 8];
            if ultra {
                find_sequences_ultra(&d, off, len, &mut st, reps, &mut seqs, &mut lits);
            } else {
                find_sequences_dfast(&d, off, len, &mut t, &mut reps, &mut seqs, &mut lits, &mut codes);
            }
            let mut r = [1u32, 4, 8];
            for q in &seqs {
                if q.match_len == 0 { continue; }
                ns += 1;
                if q.match_len <= 4 { nm3 += 1; }
                if r.contains(&q.offset) { rep += 1; }
                r = if q.offset == r[0] { r } else if q.offset == r[1] { [r[1], r[0], r[2]] } else if q.offset == r[2] { [r[2], r[0], r[1]] } else { [q.offset, r[0], r[1]] };
            }
            nl += lits.len();
            if std::env::var("BLOCKS").is_ok() && off / (256 * 1024) < 8 {
                let ms: usize = seqs.iter().filter(|q| q.match_len > 0).count();
                println!("  block {}: {} seqs, literals {:.1}%", off / (256 * 1024), ms, 100.0 * lits.len() as f64 / len as f64);
            }
            off += len;
        }
        println!("{f} {name:5}: {ns} sequences ({:.1} B/seq), literals {:.1}%, ml<=4 {:.1}%, reps {:.1}%", d.len() as f64 / ns as f64, 100.0 * nl as f64 / d.len() as f64, 100.0 * nm3 as f64 / ns as f64, 100.0 * rep as f64 / ns as f64);
    }
    for lvl in [3, 9, 19] {
        let mut enc = zstd::Encoder::new(Vec::new(), lvl).unwrap();
        enc.set_parameter(zstd::zstd_safe::CParameter::WindowLog(21)).unwrap();
        enc.write_all(&d).unwrap();
        println!("{f} zstd -{lvl}: ratio {:.3}", d.len() as f64 / enc.finish().unwrap().len() as f64);
    }
}
