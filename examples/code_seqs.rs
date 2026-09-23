// Scratch: our entropy stage on sequences from a file (litLen,
// matchLen, offset u32 triples over <file>.input, block ends as
// literal-only entries): the section sizes summed over the blocks.
use glyd::format::{BlockHeader, FLAG_COMPRESSED, MAGIC, VERSION_V8};
use glyd::v7_encode::{encode_block_with, payload_layout_of, EncScratch, Sequence, Tables};
fn main() {
    let f = std::env::args().nth(1).unwrap();
    let sq = std::fs::read(&f).unwrap();
    let d = std::fs::read(format!("{f}.input")).unwrap();
    let mut prev = Tables::none();
    let mut scratch = EncScratch::new();
    let (mut seqs, mut lits, mut payload) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pos, mut total, mut blocks) = (0usize, 0usize, 0usize);
    let mut sizes = [0u64; 5];
    let mut nseq = 0usize;
    // GLYD_MERGE=k: k of the file's blocks coded as one (128 KB blocks -> 128k KB)
    let merge: usize = std::env::var("GLYD_MERGE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut delims = 0usize;
    for t in sq.chunks_exact(12) {
        let (ll, ml, off) = (u32::from_le_bytes(t[0..4].try_into().unwrap()), u32::from_le_bytes(t[4..8].try_into().unwrap()), u32::from_le_bytes(t[8..12].try_into().unwrap()));
        lits.extend_from_slice(&d[pos..pos + ll as usize]);
        pos += (ll + ml) as usize;
        if ml == 0 {
            delims += 1;
            if delims % merge != 0 && pos < d.len() {
                // not a block end here: the trailing literals carry on
                continue;
            }
        }
        seqs.push(Sequence { lit_len: ll, match_len: ml, offset: off });
        if ml == 0 {
            payload.clear();
            encode_block_with(&seqs, &lits, 0, &mut prev, &mut scratch, false, &mut payload);
            let h = BlockHeader { magic: MAGIC, version: VERSION_V8, flags: FLAG_COMPRESSED, checksum: 0, uncompressed_len: 0, token_count: seqs.len() as u32, token_bytes: payload.len() as u32, offset_bytes: 0, extras_bytes: 0, literal_len: lits.len() as u32 };
            if let Some(l) = payload_layout_of(&h, &payload) {
                for (a, b) in sizes.iter_mut().zip(l.sub.sizes.iter()) { *a += *b as u64; }
            }
            total += payload.len() + 8;
            blocks += 1;
            nseq += seqs.len();
            seqs.clear();
            lits.clear();
        }
    }
    println!("{}: {} B in {} blocks, {} sequences; sections lit {} ll {} ml {} off {} extras {}", f.rsplit('/').next().unwrap(), total, blocks, nseq, sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]);
}
