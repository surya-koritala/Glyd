// Scratch: over a sequential max-level stream, how often blocks reuse
// the previous block's literal table and sequence tables, and the
// sections' bytes. `reuse_probe <file.glyd>`.
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let c = std::fs::read(&f).unwrap();
    let (mut cursor, mut blocks, mut raw, mut lit_reuse, mut seq_reuse, mut seq_coded) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut sizes = [0u64; 5];
    let mut small = 0usize;
    while cursor < c.len() {
        let (h, used) = glyd::format::BlockHeader::read(&c[cursor..]).unwrap();
        let end = cursor + used + h.payload_len();
        let body = &c[cursor + used..end];
        blocks += 1;
        if h.flags & glyd::format::FLAG_RAW_UNCOMPRESSED != 0 {
            raw += 1;
        } else if let Some(l) = glyd::v7_encode::payload_layout_of(&h, body) {
            if l.sub.reuse & 1 != 0 { lit_reuse += 1; }
            if l.sub.reuse & 2 != 0 { seq_reuse += 1; }
            if l.sub.coded & 0b1110 == 0b1110 { seq_coded += 1; }
            if h.uncompressed_len < 64 * 1024 { small += 1; }
            for (a, b) in sizes.iter_mut().zip(l.sub.sizes.iter()) { *a += *b as u64; }
        }
        cursor = end;
    }
    let coded = blocks - raw;
    println!("{}: {} blocks ({} raw, {} under 64 KB), {} B; literal table reused in {} of {} coded, sequence tables reused in {} ({} with all three coded); bytes per coded block: lit {} ll {} ml {} off {} extras {}",
        f.rsplit('/').next().unwrap(), blocks, raw, small, c.len(), lit_reuse, coded, seq_reuse, seq_coded,
        sizes[0] / coded.max(1) as u64, sizes[1] / coded.max(1) as u64, sizes[2] / coded.max(1) as u64, sizes[3] / coded.max(1) as u64, sizes[4] / coded.max(1) as u64);
}
