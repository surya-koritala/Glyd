// Bytes per section of the ultra output of one file (literals, ll, ml,
// off, extra), from the block sub-headers, plus literal and sequence
// counts from the block headers.
use glyd::format::{BlockHeader, HEADER_SIZE, FLAG_RAW_UNCOMPRESSED};
use glyd::v7_encode::payload_layout;
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let d = std::fs::read(format!("corpus/{f}")).unwrap();
    let mut c = Vec::new();
    glyd::compress_into_ultra(&d, &mut c);
    let (mut sizes, mut lits, mut seqs, mut blocks, mut cursor) = ([0u64; 5], 0u64, 0u64, 0u64, 0usize);
    while cursor < c.len() {
        let h: BlockHeader = unsafe { std::ptr::read_unaligned(c[cursor..].as_ptr() as *const BlockHeader) };
        let len = h.payload_len();
        blocks += 1;
        if h.flags & FLAG_RAW_UNCOMPRESSED == 0 {
            let l = payload_layout(&c[cursor + HEADER_SIZE..cursor + HEADER_SIZE + len]).unwrap();
            for i in 0..5 { sizes[i] += l.sub.sizes[i] as u64; }
            lits += h.literal_len as u64; seqs += h.token_count as u64;
        }
        cursor += HEADER_SIZE + len;
    }
    println!("{f}: {} bytes, {blocks} blocks, {lits} literals ({:.2} bits each), {seqs} seqs; sections KB: lit {} ll {} ml {} off {} extra {} ({:.1} bits/seq for codes+extra)",
        c.len(), 8.0 * sizes[0] as f64 / lits as f64, sizes[0] / 1024, sizes[1] / 1024, sizes[2] / 1024, sizes[3] / 1024, sizes[4] / 1024, 8.0 * (sizes[1] + sizes[2] + sizes[3] + sizes[4]) as f64 / seqs as f64);
}
