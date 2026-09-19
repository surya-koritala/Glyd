// What zstd -19 spends where, from its frame format: per file, literal
// bytes (regenerated) and their compressed size, sequence counts and the
// sequence sections' size; next to Glyd --ultra's parse (literal bytes,
// sequences) and output. RFC 8878 block/literals/sequences headers only.
use std::io::Write;

struct Z { lit_regen: u64, lit_comp: u64, seqs: u64, seq_bytes: u64, blocks: u64, raw: u64 }

fn parse_frame(f: &[u8]) -> Z {
    let mut z = Z { lit_regen: 0, lit_comp: 0, seqs: 0, seq_bytes: 0, blocks: 0, raw: 0 };
    let mut p = 4; // magic
    let fhd = f[p]; p += 1;
    let fcs_flag = fhd >> 6; let single = (fhd >> 5) & 1; let did = fhd & 3;
    if single == 0 { p += 1; }
    p += [0, 1, 2, 4][did as usize];
    p += match fcs_flag { 0 => if single == 1 { 1 } else { 0 }, 1 => 2, 2 => 4, _ => 8 };
    loop {
        let bh = u32::from_le_bytes([f[p], f[p + 1], f[p + 2], 0]); p += 3;
        let last = bh & 1; let btype = (bh >> 1) & 3; let bsize = (bh >> 3) as usize;
        z.blocks += 1;
        if btype == 2 {
            let b = &f[p..p + bsize];
            // Literals section header.
            let h0 = b[0]; let ltype = h0 & 3; let sf = (h0 >> 2) & 3;
            let (regen, comp, hlen) = match ltype {
                0 | 1 => {
                    let (regen, hlen) = match sf { 0 | 2 => ((h0 >> 3) as usize, 1), 1 => (((h0 >> 4) as usize) | ((b[1] as usize) << 4), 2), _ => (((h0 >> 4) as usize) | ((b[1] as usize) << 4) | ((b[2] as usize) << 12), 3) };
                    (regen, if ltype == 0 { regen } else { 1 }, hlen)
                }
                _ => {
                    let v = u32::from_le_bytes([b[0], b[1], b[2], b[3], ]) as u64 | ((b[4] as u64) << 32);
                    match sf {
                        0 | 1 => (((v >> 4) & 0x3FF) as usize, ((v >> 14) & 0x3FF) as usize, 3),
                        2 => (((v >> 4) & 0x3FFF) as usize, ((v >> 18) & 0x3FFF) as usize, 4),
                        _ => (((v >> 4) & 0x3FFFF) as usize, ((v >> 22) & 0x3FFFF) as usize, 5),
                    }
                }
            };
            z.lit_regen += regen as u64; z.lit_comp += (comp + hlen) as u64;
            let s = &b[hlen + comp..];
            let (nseq, shl) = if s[0] < 128 { (s[0] as u64, 1) } else if s[0] < 255 { ((((s[0] as u64) - 128) << 8) + s[1] as u64, 2) } else { (s[1] as u64 + ((s[2] as u64) << 8) + 0x7F00, 3) };
            z.seqs += nseq; z.seq_bytes += (s.len() - shl) as u64 + shl as u64;
        } else { z.raw += bsize as u64; }
        p += if btype == 1 { 1 } else { bsize };
        if last == 1 { break; }
    }
    z
}

fn main() {
    use glyd::v7_encode::Sequence;
    use glyd::v7_ultra::{find_sequences_ultra, UltraState};
    let mut files: Vec<_> = std::fs::read_dir("corpus").unwrap().filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.is_file() && !p.file_name().unwrap().to_string_lossy().starts_with("enwik")).collect();
    files.sort();
    let mut st = UltraState::new();
    println!("{:10} {:>9} {:>9} {:>8} | {:>9} {:>9} {:>8} | {:>9} {:>9}", "file", "z lit B", "z lit KB", "z seqs", "g lit B", "g lit KB", "g seqs", "z out KB", "g out KB");
    for p in files {
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        let d = std::fs::read(&p).unwrap();
        let mut enc = zstd::Encoder::new(Vec::new(), 19).unwrap();
        enc.write_all(&d).unwrap();
        let zf = enc.finish().unwrap();
        let z = parse_frame(&zf);
        st.clear(d.len());
        let (mut seqs, mut lits): (Vec<Sequence>, Vec<u8>) = (Vec::new(), Vec::new());
        let (mut glit, mut gseq, mut glitkb) = (0u64, 0u64, 0f64);
        let mut off = 0;
        while off < d.len() {
            let len = (d.len() - off).min(256 * 1024);
            find_sequences_ultra(&d, off, len, &mut st, [1, 4, 8], &mut seqs, &mut lits);
            glit += lits.len() as u64; gseq += seqs.iter().filter(|q| q.match_len > 0).count() as u64;
            let mut h = [0u64; 256]; for &b in &lits { h[b as usize] += 1; }
            let t = lits.len() as f64;
            glitkb += h.iter().filter(|&&c| c > 0).map(|&c| c as f64 * (t / c as f64).log2()).sum::<f64>() / 8192.0;
            off += len;
        }
        let mut g = Vec::new(); glyd::compress_into_ultra(&d, &mut g);
        println!("{name:10} {:9} {:9} {:8} | {:9} {:9.0} {:8} | {:9} {:9}", z.lit_regen, z.lit_comp / 1024, z.seqs, glit, glitkb, gseq, zf.len() / 1024, g.len() / 1024);
    }
}
