// Scratch: where the coder's bytes go against the ideal.
// `coder_probe lit <file>`: per 256 KB block, the literal Huffman coder's
// cost (its length-limited lengths) against an optimal length-limited
// code (package-merge, 11 bits) and the entropy bound.
// `coder_probe seq <dump>`: zstd's sequences (`code_seqs` format) coded
// by our block coder, each section against its entropy bound.
use glyd::huffman::MAX_CODE_LEN;
use glyd::v7_encode::{encode_block_with, payload_layout_of, EncScratch, Sequence, Tables};
use glyd::v7_format::{ll_code, ml_code, Reps};

fn entropy_bits(hist: &[u64]) -> f64 {
    let n: u64 = hist.iter().sum();
    if n == 0 {
        return 0.0;
    }
    hist.iter().filter(|&&c| c > 0).map(|&c| c as f64 * ((n as f64) / (c as f64)).log2()).sum()
}

/// Optimal code lengths under a length limit (package-merge).
fn package_merge(hist: &[u64; 256], limit: usize) -> [u8; 256] {
    let syms: Vec<usize> = (0..256).filter(|&s| hist[s] > 0).collect();
    let mut lengths = [0u8; 256];
    let n = syms.len();
    if n <= 1 {
        for &s in &syms {
            lengths[s] = 1;
        }
        return lengths;
    }
    // An item is (weight, the leaves it contains as a bitmap over syms).
    let mut level: Vec<(u64, Vec<u16>)> = syms.iter().map(|&s| (hist[s], vec![s as u16])).collect();
    level.sort_by_key(|x| x.0);
    let leaves: Vec<(u64, Vec<u16>)> = level.clone();
    for _ in 1..limit {
        let mut next: Vec<(u64, Vec<u16>)> = Vec::with_capacity(level.len() / 2 + n);
        for pair in level.chunks_exact(2) {
            let mut v = pair[0].1.clone();
            v.extend_from_slice(&pair[1].1);
            next.push((pair[0].0 + pair[1].0, v));
        }
        next.extend(leaves.iter().cloned());
        next.sort_by_key(|x| x.0);
        level = next;
    }
    for item in level.iter().take(2 * n - 2) {
        for &s in &item.1 {
            lengths[s as usize] += 1;
        }
    }
    lengths
}

fn main() {
    let mode = std::env::args().nth(1).unwrap();
    let f = std::env::args().nth(2).unwrap();
    if mode == "lit" {
        let d = std::fs::read(&f).unwrap();
        let (mut raw, mut ent, mut ours, mut opt, mut ours_raw, mut opt_raw) = (0u64, 0f64, 0u64, 0u64, 0usize, 0usize);
        for block in d.chunks(256 << 10) {
            let mut h = [0u64; 256];
            for &b in block {
                h[b as usize] += 1;
            }
            let ours_len = glyd::huff8::lengths_for(&h);
            let opt_len = package_merge(&h, MAX_CODE_LEN as usize);
            let bits = |l: &[u8; 256]| (0..256).map(|s| h[s] * l[s] as u64).sum::<u64>();
            let ours_b = bits(&ours_len) / 8 + glyd::huffman::packed_lengths_v8_size(&ours_len) as u64;
            let opt_b = bits(&opt_len) / 8 + glyd::huffman::packed_lengths_v8_size(&opt_len) as u64;
            raw += block.len() as u64;
            ent += entropy_bits(&h) / 8.0;
            // The coder's rule: coded only when it saves 2%.
            if (ours_b as usize) + block.len() / 50 < block.len() { ours += ours_b } else { ours += block.len() as u64; ours_raw += 1 }
            if (opt_b as usize) < block.len() { opt += opt_b } else { opt += block.len() as u64; opt_raw += 1 }
        }
        println!("{}: raw {} B, entropy {:.0} B, our lengths {} B ({} blocks raw), package-merge {} B ({} blocks raw): ours {:+.2}% vs optimal", f.rsplit('/').next().unwrap(), raw, ent, ours, ours_raw, opt, opt_raw, (ours as f64 / opt as f64 - 1.0) * 100.0);
        return;
    }
    // Sequences: our sections against their entropy bounds, per section.
    let sq = std::fs::read(&f).unwrap();
    let d = std::fs::read(format!("{f}.input")).unwrap();
    let mut prev = Tables::none();
    let mut scratch = EncScratch::new();
    let (mut seqs, mut lits, mut payload) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pos, mut delims) = (0usize, 0usize);
    let merge: usize = std::env::var("GLYD_MERGE").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut actual = [0u64; 5];
    let mut ideal = [0f64; 5];
    let (mut blocks, mut total) = (0usize, 0u64);
    let (mut one_table, mut two_tables) = (0u64, 0u64);
    for t in sq.chunks_exact(16) {
        let (ll, ml, off) = (u32::from_le_bytes(t[0..4].try_into().unwrap()), u32::from_le_bytes(t[4..8].try_into().unwrap()), u32::from_le_bytes(t[8..12].try_into().unwrap()));
        lits.extend_from_slice(&d[pos..pos + ll as usize]);
        pos += (ll + ml) as usize;
        if ml == 0 {
            delims += 1;
            if delims % merge != 0 && pos < d.len() {
                continue;
            }
        }
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset: off });
        if ml == 0 {
            // Ideal: entropy of each code alphabet plus the extra bits.
            let mut hl = [0u64; 64];
            let mut hm = [0u64; 64];
            let mut ho = [0u64; 64];
            let mut extra = 0u64;
            let mut reps = Reps::new();
            for (i, q) in seqs.iter().enumerate() {
                let (lc, lb, _) = ll_code(q.lit_len);
                let (mc, mb, _) = ml_code(q.match_len.max(3));
                hl[lc as usize] += 1;
                hm[mc as usize] += 1;
                extra += lb as u64 + mb as u64;
                if i + 1 < seqs.len() {
                    let (oc, ob, _) = reps.code_for(q.offset, q.lit_len == 0);
                    ho[oc as usize] += 1;
                    extra += ob as u64;
                }
            }
            let mut hlit = [0u64; 256];
            for &b in &lits {
                hlit[b as usize] += 1;
            }
            ideal[0] += entropy_bits(&hlit) / 8.0;
            // One table for the block's literals against one per half
            // (the second half's table paid for): the gain of a second
            // literal table per block, summed in `two_tables`.
            {
                let one = glyd::huff8::lengths_for(&hlit);
                let cost = |h: &[u64; 256], l: &[u8; 256]| (0..256).map(|s| h[s] * l[s] as u64).sum::<u64>() / 8 + glyd::huffman::packed_lengths_v8_size(l) as u64;
                let c1 = cost(&hlit, &one);
                let mid = lits.len() / 2;
                let (mut ha, mut hb) = ([0u64; 256], [0u64; 256]);
                for &b in &lits[..mid] { ha[b as usize] += 1; }
                for &b in &lits[mid..] { hb[b as usize] += 1; }
                let c2 = cost(&ha, &glyd::huff8::lengths_for(&ha)) + cost(&hb, &glyd::huff8::lengths_for(&hb));
                two_tables += c1.saturating_sub(c2.min(c1));
                one_table += c1;
            }
            ideal[1] += entropy_bits(&hl) / 8.0;
            ideal[2] += entropy_bits(&hm) / 8.0;
            ideal[3] += entropy_bits(&ho) / 8.0;
            ideal[4] += extra as f64 / 8.0;
            payload.clear();
            encode_block_with(&seqs, &lits, 0, &mut prev, &mut scratch, false, &mut payload);
            let h = glyd::format::BlockHeader { magic: glyd::format::MAGIC, version: glyd::format::VERSION_V8, flags: glyd::format::FLAG_COMPRESSED, checksum: 0, uncompressed_len: 0, token_count: seqs.len() as u32, token_bytes: payload.len() as u32, offset_bytes: 0, extras_bytes: 0, literal_len: lits.len() as u32 };
            if let Some(l) = payload_layout_of(&h, &payload) {
                for (a, b) in actual.iter_mut().zip(l.sub.sizes.iter()) {
                    *a += *b as u64;
                }
            }
            total += payload.len() as u64;
            blocks += 1;
            seqs.clear();
            lits.clear();
        }
    }
    let names = ["lit", "ll", "ml", "off", "extras"];
    print!("{}: {} blocks, total {} B (sections {} B, framing {} B);", f.rsplit('/').next().unwrap(), blocks, total, actual.iter().sum::<u64>(), total - actual.iter().sum::<u64>());
    for i in 0..5 {
        print!(" {} {} B vs ideal {:.0} ({:+.1}%);", names[i], actual[i], ideal[i], (actual[i] as f64 / ideal[i] - 1.0) * 100.0);
    }
    println!();
    println!("  a second literal table per block would save {} B of {} ({:.2}%)", two_tables, one_table, two_tables as f64 * 100.0 / one_table.max(1) as f64);
}
